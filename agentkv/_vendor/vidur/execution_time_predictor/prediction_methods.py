# Inference methods from the pinned Vidur SklearnExecutionTimePredictor.
from __future__ import annotations
from agentkv._vendor.vidur.entities import Batch, ExecutionTime
from agentkv._vendor.vidur.execution_time_predictor.sklearn_execution_time_predictor_batch import SklearnExecutionTimePredictorBatch

class PredictionMethods:
    def get_batch_execution_time(
        self, batch: Batch, pipeline_stage: int
    ) -> ExecutionTime:
        sklearn_batch = SklearnExecutionTimePredictorBatch.from_batch(
            batch,
            self._config.kv_cache_prediction_granularity,
            self._config.prefill_chunk_size_prediction_granularity,
        )
        return self._get_execution_time(sklearn_batch, pipeline_stage)

    def get_execution_time(
        self,
        execution_time_predictor_requests: List[ExecutionTimePredictorRequest],
        pipeline_stage: int,
    ) -> ExecutionTime:
        sklearn_batch = SklearnExecutionTimePredictorBatch(
            execution_time_predictor_requests,
            self._config.kv_cache_prediction_granularity,
            self._config.prefill_chunk_size_prediction_granularity,
        )
        return self._get_execution_time(sklearn_batch, pipeline_stage)

    def _get_execution_time(
        self, batch: SklearnExecutionTimePredictorBatch, pipeline_stage: int
    ) -> ExecutionTime:
        if pipeline_stage == self._replica_config.num_pipeline_stages - 1:
            pipeline_parallel_communication_time = 0
        else:
            pipeline_parallel_communication_time = (
                self._get_pipeline_parallel_communication_time(batch)
            )

        if self._replica_config.tensor_parallel_size == 1:
            tensor_parallel_communication_time = 0
        else:
            tensor_parallel_communication_time = (
                self._get_tensor_parallel_communication_time(batch)
            )

        return ExecutionTime(
            self._num_layers_per_pipeline_stage,
            self._get_attention_rope_execution_time(batch),
            self._get_attention_kv_cache_save_execution_time(batch),
            self._get_attention_decode_execution_time(batch),
            self._get_attention_prefill_execution_time(batch),
            self._get_attention_layer_pre_proj_execution_time(batch),
            self._get_attention_layer_post_proj_execution_time(batch),
            self._get_mlp_layer_up_proj_execution_time(batch),
            self._get_mlp_layer_down_proj_execution_time(batch),
            self._get_mlp_layer_act_execution_time(batch),
            self._get_attn_norm_layer_act_execution_time(batch),
            self._get_mlp_norm_layer_act_execution_time(batch),
            self._get_add_layer_act_execution_time(batch),
            tensor_parallel_communication_time,
            pipeline_parallel_communication_time,
            self._get_schedule_time(batch),
            self._get_sampler_e2e_time(batch),
            self._get_prepare_inputs_e2e_time(batch),
            self._get_process_model_outputs_time(batch),
            self._get_ray_comm_time(batch),
        )

    def _get_attention_layer_pre_proj_execution_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        return self._predictions["attn_pre_proj"][(batch.total_num_tokens_rounded,)]

    def _get_attention_layer_post_proj_execution_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        return self._predictions["attn_post_proj"][(batch.total_num_tokens_rounded,)]

    def _get_mlp_layer_up_proj_execution_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        return self._predictions["mlp_up_proj"][(batch.total_num_tokens_rounded,)]

    def _get_mlp_layer_down_proj_execution_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        return self._predictions["mlp_down_proj"][(batch.total_num_tokens_rounded,)]

    def _get_mlp_layer_act_execution_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        return self._predictions["mlp_act"][(batch.total_num_tokens_rounded,)]

    def _get_attn_norm_layer_act_execution_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        return self._predictions["input_layernorm"][(batch.total_num_tokens_rounded,)]

    def _get_mlp_norm_layer_act_execution_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        if not self._model_config.post_attn_norm:
            return 0

        return self._predictions["post_attention_layernorm"][
            (batch.total_num_tokens_rounded,)
        ]

    def _get_add_layer_act_execution_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        return self._predictions["add"][(batch.total_num_tokens_rounded,)]

    def _get_tensor_parallel_communication_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        return (
            self._predictions["all_reduce"][(batch._total_num_tokens_rounded,)]
            + self._config.nccl_cpu_launch_overhead_ms
            + self._config.nccl_cpu_skew_overhead_per_device_ms
            * self._replica_config.tensor_parallel_size**1.25
        )

    def _get_pipeline_parallel_communication_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        try:
            return self._predictions["send_recv"][(batch.total_num_tokens_rounded,)]
        except KeyError as e:
            logger.error(f"Failed to get send_recv prediction for batch {batch}")
            raise e

    def _get_attention_rope_execution_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        return self._predictions["attn_rope"][(batch.total_num_tokens_rounded,)]

    def _get_attention_kv_cache_save_execution_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        # don't use round up to the nearest multiple of 8 here, because we want to
        # predict the execution time for the exact number of tokens
        return self._predictions["attn_kv_cache_save"][(batch.total_num_tokens,)]

    def _get_attention_decode_execution_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        if batch.decode_batch_size == 0:
            return 0

        return self._predictions["attn_decode"][
            (batch.decode_batch_size, batch.decode_avg_kv_cache_size)
        ] * (
            1
            + self._attention_decode_batching_overhead_fraction
            * int(batch.decode_batch_size > 1)
        )

    def _get_attention_prefill_execution_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        if batch.prefill_batch_size == 0:
            return 0

        return self._predictions["attn_prefill"][
            (batch.prefill_agg_kv_cache_size, batch.prefill_agg_chunk_size)
        ] * (
            1
            + self._attention_prefill_batching_overhead_fraction
            * int(batch.prefill_batch_size > 1)
        )

    def _get_schedule_time(self, batch: SklearnExecutionTimePredictorBatch) -> float:
        if self._config.skip_cpu_overhead_modeling:
            return 0

        return self._predictions["schedule"][(batch.size,)]

    def _get_sampler_e2e_time(self, batch: SklearnExecutionTimePredictorBatch) -> float:
        if self._config.skip_cpu_overhead_modeling:
            return 0

        return self._predictions["sampler_e2e"][(batch.size,)]

    def _get_prepare_inputs_e2e_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        if self._config.skip_cpu_overhead_modeling:
            return 0

        return self._predictions["prepare_inputs_e2e"][(batch.size,)]

    def _get_process_model_outputs_time(
        self, batch: SklearnExecutionTimePredictorBatch
    ) -> float:
        if self._config.skip_cpu_overhead_modeling:
            return 0

        return self._predictions["process_model_outputs"][(batch.size,)]

    def _get_ray_comm_time(self, batch: SklearnExecutionTimePredictorBatch) -> float:
        if self._config.skip_cpu_overhead_modeling:
            return 0

        return self._predictions["ray_comm_time"][(batch.size,)]
