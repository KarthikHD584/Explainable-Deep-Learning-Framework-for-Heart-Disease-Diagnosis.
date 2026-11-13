Put the deployment bundle here (output of `export_for_deployment.py`):

    models/
      model_config.json
      model_run0.keras ... model_run4.keras   (or a single model.keras)
      requirements.lock.txt

The .keras files are ~16 MB each; track them with Git LFS (`git lfs install`; `.gitattributes` is set up).
