import torch


def save_checkpoint(
    model,
    optimizer,
    step,
    path,
):
    checkpoint = {

        # 模型参数
        "model_state_dict":
            model.state_dict(),

        # optimizer状态
        "optimizer_state_dict":
            optimizer.state_dict(),

        # 当前训练步数
        "step":
            step,
    }

    torch.save(
        checkpoint,
        path,
    )

def load_checkpoint(
    path,
    model,
    optimizer,
):

    checkpoint = torch.load(
        path,
        map_location="cpu"
    )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )


    step = checkpoint["step"]


    return step