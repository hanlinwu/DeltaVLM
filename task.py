import torch.distributed as dist
import logging
import torch
import os
from model.blip2_qformer import Blip2Qformer
from model.blip2OPT import Blip2OPT
from model.blip2_vicua import Blip2VicunaInstruct
from logger import MetricLogger, SmoothedValue
from dataset import COCOCapBuilder
from utils import prepare_sample, get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized, is_convertible_to_int


class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"
        

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg
        if model_config.arch == "blip2":
            model_cls = Blip2Qformer
        elif model_config.arch == "blip2_opt":
            model_cls = Blip2OPT
        elif model_config.arch == "instruct_vicuna7b":
            model_cls = Blip2VicunaInstruct
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg  # changechat_105k

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]
            builder = COCOCapBuilder(
                dataset_config)  # get_builder_class(name):"builder_name_mapping":builder->caption_builder->COCOCapBuilder(BaseDatasetBuilder)
            # dataset_config
            dataset = builder.build_datasets()

            datasets[name] = dataset

        return datasets

    def train_step(self, model, samples):
        output = model(samples)
        loss_dict = {}
        for k, v in output.items():
            if "loss" in k:
                loss_dict[k] = v
        return output["loss"], loss_dict

    def valid_step(self, model, samples):
        # raise NotImplementedError
        # import pdb;pdb.set_trace();
        # 增加captaining部分代码
        results = []
        # run_cfg = slf.cfg.run_cfg
        # captions = model.generate(
        captions = model.predict_answers(
            samples,
            use_nucleus_sampling=False,
            # num_beams=self.num_beams,
            # max_length=self.max_len,
            # min_length=self.min_len,
            # repetition_penalty=self.repetition_penalty,

            # length_penalty=self.length_penalty,
            # top_p=self.top_p,
            # temperature=self.temperature,
        )
        img_ids = samples['image_id']
        # print(img_ids)
        for caption, img_id in zip(captions, img_ids):
            # not all img_ids are ints
            img_id = int(img_id) if is_convertible_to_int(img_id) else img_id
            # print(img_id)
            # if self.img_ids and img_id not in self.img_ids: # only include specified img_ids if specified
            #     continue
            results.append({"caption": caption, "image_id": img_id})
            # results.append(caption)

        return results

    def before_training(self, model, dataset, **kwargs):
        model.before_training(dataset=dataset, task_type=type(self))

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True):
        # 缩小测试集
        # from torch.utils.data import Subset, DataLoader
        # subset_size = [i for i in range(50)]
        # data_loader = Subset(data_loader, subset_size)
        # data_loader = DataLoader(data_loader, batch_size=32, shuffle=False)
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def train_epoch(
            self,
            epoch,
            model,
            data_loader,
            optimizer,
            lr_scheduler,
            scaler=None,
            cuda_enabled=False,
            log_freq=50,
            accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=len(data_loader),
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def train_iters(
            self,
            epoch,
            start_iters,
            iters_per_inner_epoch,
            model,
            data_loader,
            optimizer,
            lr_scheduler,
            scaler=None,
            cuda_enabled=False,
            log_freq=50,
            accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
            self,
            epoch,
            iters_per_epoch,
            model,
            data_loader,
            optimizer,
            lr_scheduler,
            scaler=None,
            start_iters=None,
            log_freq=50,
            cuda_enabled=False,
            accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            ## notify model that sample is empty (error occured)
            if not isinstance(samples, dict):
                samples = {"is_empty": True}

            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, loss_dict = self.train_step(model=model, samples=samples)
                loss /= accum_grad_iters  # TODO: not affect loss_dict values for logging

            # after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(**loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file


class ImageTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    def after_evaluation(self, val_result, split_name, epoch, output_dir, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=output_dir,
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="image_id",
        )

        # if self.report_metric:
        #     metrics = self._report_metrics(
        #         eval_result_file=eval_result_file, split_name=split_name
        #     )
        # else:
        #     metrics = {"agg_metrics": 0.0}
        # metrics = {"agg_metrics": 0.0}

        # return metrics

    # @classmethod
    # def setup_task(cls, cfg):
    #     run_cfg = cfg.run_cfg
    #     sample_id_key = run_cfg.get("sample_id_key", "image_id")
    #     return cls(
    #         sample_id_key=sample_id_key,
    #     )
    # def evaluation(self, model, data_loader, cuda_enabled=True):
    #     pass