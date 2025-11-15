import json

def main() -> None:
    info = {}

    # PyTorch
    try:
        import torch  # type: ignore
        info["torch"] = {
            "version": getattr(torch, "__version__", "unknown"),
            "cuda_available": bool(getattr(torch.cuda, "is_available", lambda: False)()),
            "device_count": int(getattr(torch.cuda, "device_count", lambda: 0)()),
        }
    except Exception as e:  # pragma: no cover
        info["torch_error"] = str(e)

    # TensorFlow
    try:
        import tensorflow as tf  # type: ignore
        gpus = [d.name for d in tf.config.list_physical_devices("GPU")]
        info["tensorflow"] = {
            "version": getattr(tf, "__version__", "unknown"),
            "gpus": gpus,
            "built_with_cuda": bool(getattr(tf.test, "is_built_with_cuda", lambda: False)()),
        }
    except Exception as e:  # pragma: no cover
        info["tensorflow_error"] = str(e)

    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()


