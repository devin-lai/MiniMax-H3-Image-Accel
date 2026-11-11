"""Start the upstream CLI without eager imports in multiprocessing children."""
if __name__ == "__main__":
    from sglang.multimodal_gen.runtime.entrypoints.cli.main import main

    main()
