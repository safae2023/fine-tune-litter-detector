import openvino as ov

print("Converting ONNX to OpenVINO...")

try:
    # Convert ONNX to OpenVINO
    ov_model = ov.convert_model("Fast_RCNN_detection_model.onnx")
    print("Model converted successfully!")
    
    # Save as IR format (creates .xml + .bin files)
    ov.save_model(ov_model, "Fast_RCNN_detection_model.xml")
    print("Model saved as Fast_RCNN_detection_model.xml")
    
    print("Conversion completed! You should now have:")
    print(" Fast_RCNN_detection_model.xml")
    print("Fast_RCNN_detection_model.bin")
    
except Exception as e:
    print(f" Error during conversion: {e}")
    import traceback
    traceback.print_exc()
