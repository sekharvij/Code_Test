import argparse
import pandas as pd
import joblib

def run_inference(model_path, input_data, output_path):
    # Load model
    model = joblib.load(model_path)
    print("Model loaded")

    # Load input
    df = pd.read_csv(input_data)
    print(f"📄 Input data: {df.shape}")

    # Predict
    predictions = model.predict(df)
    df['prediction'] = predictions

    # Save output
    df.to_csv(output_path, index=False)
    print(f"📝 Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using saved model")
    parser.add_argument('--model', required=True, help='Path to saved model (.pkl)')
    parser.add_argument('--input', required=True, help='Input features CSV')
    parser.add_argument('--output', required=True, help='CSV to save predictions')

    args = parser.parse_args()
    run_inference(args.model, args.input, args.output)
