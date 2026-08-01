# Remote Support

This folder contains a remote-friendly training and evaluation path for the BraTS 2020 2D segmentation project.

Goals:
- Keep the current local workflow unchanged.
- Add a separate entry point for remote GPU platforms such as Tesla V100 time-limited sessions.
- Support resume training from `last_model.pth`.
- Write outputs to a separate artifacts directory by default.

Recommended upload set for a remote platform:
- `remote_support/`
- `src/`
- `data_processed/`
- `requirements.txt`

Default remote config:
- `remote_support/configs/v100_remote_config.yaml`

Default remote outputs:
- `artifacts_remote/experiments/<experiment_name>/`

Typical commands:

```powershell
python remote_support/train_remote.py --experiment unet_bce_dice
python remote_support/train_remote.py --experiment unet_bce_dice --resume
python remote_support/evaluate_remote.py --experiment unet_bce_dice
```

If the remote platform gives only 2 hours per session:
- Start one experiment.
- If the session ends early, reconnect and rerun the same command with `--resume`.
- The remote trainer will continue from `last_model.pth` and keep the existing history.

Useful overrides:

```powershell
python remote_support/train_remote.py --experiment attention_unet_bce_dice --batch-size 16 --num-workers 4
python remote_support/evaluate_remote.py --experiment attention_unet_bce_dice --artifacts-root /root/project/artifacts_remote
```
