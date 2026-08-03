import torch
import torch.nn as nn
import torchvision.models as models

CLASS_NAMES = [
    "path_free",             # 0: Open lane ahead
    "obstacle_blocked",      # 1: Blocked by wall or object
    "parking_spot_left",     # 2: Open spot on left side
    "parking_spot_right",    # 3: Open spot on right side
    "parking_spot_occupied"  # 4: Spot occupied by another vehicle/object
]

class ParkingNet(nn.Module):
    """
    Lightweight Transfer Learning model based on MobileNetV2 (or ResNet18)
    designed specifically for real-time inference on NVIDIA Jetson Nano.
    """
    def __init__(self, num_classes=len(CLASS_NAMES), backbone="mobilenet_v2", pretrained=True):
        super(ParkingNet, self).__init__()
        self.num_classes = num_classes
        self.backbone_type = backbone

        if backbone == "mobilenet_v2":
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            # Replace final classification head
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features, num_classes)
            )
        elif backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        return self.backbone(x)

    def export_onnx(self, output_path="parking_net.onnx"):
        """Export PyTorch model to ONNX for TensorRT optimization on Jetson."""
        self.eval()
        dummy_input = torch.randn(1, 3, 224, 224, device=next(self.parameters()).device)
        torch.onnx.export(
            self,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=11
        )
        print(f"Exported model to ONNX format at {output_path}")
