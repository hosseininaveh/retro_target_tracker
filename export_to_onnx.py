import torch
import argparse
from pathlib import Path

# Import your models
from models.heatmap_tracker import HeatmapTracker
from models.coord_regressor import CoordRegressor
from models.shared_blocks import SubpixelDecoder, AttentionRefinement

def export_heatmap_tracker(pth_path, onnx_path, input_shape=(1, 3, 256, 256)):
    """Export HeatmapTracker model to ONNX"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HeatmapTracker().to(device)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['heatmap'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'heatmap': {0: 'batch_size'}
        }
    )
    print(f"Successfully exported HeatmapTracker to {onnx_path}")

def export_coord_regressor(pth_path, onnx_path, input_shape=(1, 3, 256, 256)):
    """Export CoordRegressor model to ONNX"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CoordRegressor().to(device)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['coordinates'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'coordinates': {0: 'batch_size'}
        }
    )
    print(f"Successfully exported CoordRegressor to {onnx_path}")

def export_subpixel_decoder(onnx_path, input_shape=(1, 1, 256, 256)):
    """Export SubpixelDecoder module to ONNX"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SubpixelDecoder().to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['heatmap'],
        output_names=['y_coords', 'x_coords'],
        dynamic_axes={
            'heatmap': {0: 'batch_size'},
            'y_coords': {0: 'batch_size'},
            'x_coords': {0: 'batch_size'}
        }
    )
    print(f"Successfully exported SubpixelDecoder to {onnx_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export models to ONNX format")
    parser.add_argument("--model-type", choices=["heatmap", "coord", "decoder"], required=True,
                       help="Type of model to export")
    parser.add_argument("--input-path", type=str, required=False,
                       help="Path to input .pth file (not needed for decoder)")
    parser.add_argument("--output-path", type=str, required=True,
                       help="Path to save ONNX file")
    parser.add_argument("--input-shape", type=int, nargs=4, default=[1, 3, 256, 256],
                       help="Input shape as batch, channels, height, width")
    
    args = parser.parse_args()
    
    # Create output directory if needed
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if args.model_type == "heatmap":
        if not args.input_path:
            raise ValueError("--input-path is required for heatmap model")
        export_heatmap_tracker(args.input_path, args.output_path, tuple(args.input_shape))
    elif args.model_type == "coord":
        if not args.input_path:
            raise ValueError("--input-path is required for coord model")
        export_coord_regressor(args.input_path, args.output_path, tuple(args.input_shape))
    elif args.model_type == "decoder":
        export_subpixel_decoder(args.output_path, tuple(args.input_shape))
