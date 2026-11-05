"""Quick check: build the model, add ECA + GC, compare parameter counts, run one forward pass."""
import torch
from ultralytics.nn.tasks import DetectionModel

from attention_modules import add_attention

base = DetectionModel("yolov8n.yaml", nc=12, verbose=False)
yza = DetectionModel("yolov8n.yaml", nc=12, verbose=False)
indices = add_attention(yza)

count = lambda m: sum(p.numel() for p in m.parameters())
print("layers with ECA + GC added:", indices)
print("baseline parameters:", count(base))
print("Y-ZA parameters:    ", count(yza))

x = torch.zeros(1, 3, 640, 640)
yza.eval()
with torch.no_grad():
    out = yza(x)
print("forward pass OK, output type:", type(out).__name__)
