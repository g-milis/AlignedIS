def undetectable_exp_pipeline(output_path, model_str, reweight_type, dataset_name, context_length=1, max_generations=None):
    import os
    from torch.multiprocessing import Process, Queue, Event
    from .common import set_spawn, get_num_gpus

    set_spawn()
    num_gpus = get_num_gpus()

    print(f"Starting queue with {num_gpus} GPUs")

    tq = Queue(maxsize=num_gpus)
    tqe = Event()
    rq = Queue()
    rqe = Event()

    from .common import batched_wp_task_worker, spiritlm_worker, speechgpt_worker
    from . import get_in_ds_undetectable_exp

    task_worker_ = Process(
        target=batched_wp_task_worker,
        args=(tq,),
        kwargs={
            "get_in_ds": get_in_ds_undetectable_exp,
            "reweight_type": reweight_type,
            "dataset_name": dataset_name,
            "batch_size": 1,
            "context_length": context_length,
            "max_generations": max_generations,
            "model_str": model_str
        },
    )

    if "spirit-lm" in model_str:
        transformer_worker = spiritlm_worker
    elif "SpeechGPT" in model_str:
        transformer_worker = speechgpt_worker
    else:
        print('Unknown model_str:', model_str)
        raise NotImplementedError
    
    save_dir = os.path.join(os.path.dirname(output_path), "wav")
    os.makedirs(save_dir, exist_ok=True)

    gpu_workers = [
        Process(
            target=transformer_worker,
            args=(tq, tqe, rq, i),
            kwargs={
                "model_str": model_str,
                "save_dir": save_dir
            }
            # kwargs={
            #     #  "model_str": "meta-llama/Llama-2-7b-chat-hf",
            #     "model_str": "daryl149/llama-2-7b-chat-hf",
            #     "decoder_only": True,
            #     "generation_kwargs": {
            #         "max_new_tokens": 512,
            #         "temperature": 1.0,
            #     },
            #     "tokenization_kwargs": {
            #         "task_template": "Help me complete the following text with at least 500 words:\n{input}",
            #         "max_length": 3072,
            #     },
            # },
        )
        for i in range(num_gpus)
    ]

    from .common import simple_store_worker

    store_worker = Process(
        target=simple_store_worker, args=(output_path, rq, rqe)
    )

    task_worker_.start()
    for w in gpu_workers:
        w.start()
    store_worker.start()

    task_worker_.join()
    assert task_worker_.exitcode == 0, "Bad task worker exit"
    tqe.set()

    for w in gpu_workers:
        w.join()
        assert w.exitcode == 0, "Bad GPU worker exit"
    rqe.set()

    store_worker.join()
    assert store_worker.exitcode == 0, "Bad store worker exit"
