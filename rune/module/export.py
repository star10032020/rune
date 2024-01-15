from ultralytics import YOLO



def main():
    # Create a new YOLO model from scratch
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('/workspaces/vscode_ros2_workspace/src/rune/rune/best.pt')

    # Perform object detection on an image using the model
    #results = model('https://ultralytics.com/images/bus.jpg')
    # Export the model to ONNX format
    success = model.export(format='engine')
    print("model has maken")


if __name__ == "__main__":
    main()                                