"""Split ONNX into sub-models for each output, convert with onnx2ncnn, merge params."""
import onnx
import subprocess
from pathlib import Path

out_dir = Path(r"F:\openhanako\yolov8-seg\ncnn_final")
onnx_file = out_dir / "yolov8n-seg_v2.onnx"
onnx2ncnn = r"F:\openhanako\yolov8-seg\ncnn\ncnn-20240410-windows-vs2022\x64\bin\onnx2ncnn.exe"

m = onnx.load(str(onnx_file))
onnx.checker.check_model(m)

# Get existing outputs
for out in m.graph.output:
    print(f"Output: {out.name}")

# Extract subgraph for output1 (proto)
# We need to create a model that only has output1 as output
# Remove output0 from the graph outputs
m.graph.output.pop(0)  # Remove output0, keep only output1

# Save sub-model
sub_onnx = out_dir / "yolov8n-seg_proto.onnx"
onnx.save(m, str(sub_onnx), save_as_external_data=False)
print(f"\nSaved proto ONNX to {sub_onnx.name}")

# Verify
m2 = onnx.load(str(sub_onnx))
for out in m2.graph.output:
    dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f"Proto output: {out.name} {dims}")

# Convert proto sub-model with onnx2ncnn
result = subprocess.run(
    [onnx2ncnn, str(sub_onnx), str(out_dir / "proto.param"), str(out_dir / "proto.bin")],
    capture_output=True, text=True, cwd=str(out_dir)
)
print(f"onnx2ncnn proto: {result.returncode}")
if result.stderr:
    print(result.stderr[:500])

# Read proto param
proto_param = out_dir / "proto.param"
if proto_param.exists():
    lines = proto_param.read_text().strip().split("\n")
    print(f"\nProto param has {len(lines)} lines")
    # Find the output1 line and append it to our main param
    # The last line should be the output
    last_line = lines[-1].strip()
    print(f"Last line: {last_line[:200]}")
    
    # Read main param
    main_param = out_dir / "yolov8n-seg.param"
    main_lines = main_param.read_text().strip().split("\n")
    
    # The main param needs to include all proto layers.
    # Strategy: insert proto layers AFTER the main model layers but BEFORE the final output0
    # Actually, the proto path is parallel to the main path.
    # We need to insert proto layers (from the proto param) before the final output
    
    # Find proto-specific layers (only those not already in main param)
    # Get layer names from main param
    main_layers = set()
    for line in main_lines:
        parts = line.split()
        if len(parts) >= 2:
            main_layers.add(parts[1])
    
    # Find unique proto layers (excluding the final output line)
    proto_layers = lines[:-1]  # All except last line (output1)
    proto_output_line = lines[-1]
    
    # Add only unique proto layers
    new_proto_layers = []
    for line in proto_layers:
        parts = line.split()
        if len(parts) >= 2 and parts[1] not in main_layers:
            new_proto_layers.append(line)
    
    print(f"Unique proto layers to add: {len(new_proto_layers)}")
    
    # Insert proto layers before the final output0 line
    merged_lines = main_lines[:-1] + new_proto_layers + [main_lines[-1]]
    
    # Now modify the last output line to rename output0 properly
    # The original output0 line from main param ends with "output0"
    # Add the proto output line, renaming output1's source
    merged_lines = main_lines[:-1] + new_proto_layers + [main_lines[-1], proto_output_line]
    
    # Check: replace any duplicate output names
    # Note: proto output must be named output1
    merged_lines[-1] = merged_lines[-1].replace(" output0", " output1")
    
    # Update layer count in header
    header = merged_lines[0]
    parts = header.split()
    # The format is: magic_number layer_count blob_count
    # Update layer_count
    old_count = int(parts[1])
    new_count = old_count + len(new_proto_layers) + 1  # +1 for the output1 line
    parts[1] = str(new_count)
    merged_lines[0] = " ".join(parts)
    
    merged_param = out_dir / "yolov8n-seg.param"
    merged_param.write_text("\n".join(merged_lines))
    print(f"\nMerged param: {len(merged_lines)} lines (was {len(main_lines)})")
    print(f"Last 3 lines:")
    for line in merged_lines[-3:]:
        print(f"  {line[:150]}")
else:
    print("Proto param not found!")
