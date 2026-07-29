from pathlib import Path
import torch
def model_saver (model,
                 target_dir,
                 model_file_name):
    target_path = Path(target_dir)
    target_path.mkdir(parents=True,
                      exist_ok=True)
    assert model_file_name.endswith(".pth") or model_file_name.endswith(".pt"),"put .pth ot .pt"
    model_save_path  = target_path / model_file_name
    print(f"saving model to {model_save_path}")
    torch.save(model.state_dict(), model_save_path)
