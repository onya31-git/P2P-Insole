import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def check_scaler_consistency(scaler_path, sample_data):
    print("🧪 Checking scaler consistency...")
    scaler = joblib.load(scaler_path)
    transformed = scaler.transform(sample_data)
    inversed = scaler.inverse_transform(transformed)
    diff = np.abs(sample_data - inversed)
    max_diff = np.max(diff)
    print(f"Max diff after inverse transform: {max_diff:.6f}")
    if max_diff < 1e-5:
        print("✅ Scaler consistency: OK")
    else:
        print("⚠️ Scaler consistency: FAILED")

def analyze_skeleton_statistics(skeleton_csv_path):
    print("📊 Analyzing predicted skeleton output...")
    df = pd.read_csv(skeleton_csv_path)
    stats = df.describe()
    print(stats)
    stats.to_csv("debug_skeleton_statistics.csv")

def analyze_motion_dynamics(skeleton_csv_path):
    print("📈 Plotting motion dynamics (velocity & acceleration)...")
    df = pd.read_csv(skeleton_csv_path).values
    velocity = np.diff(df, axis=0)
    acceleration = np.diff(velocity, axis=0)
    velocity_norm = np.linalg.norm(velocity, axis=1)
    acceleration_norm = np.linalg.norm(acceleration, axis=1)

    plt.plot(velocity_norm, label="Velocity norm")
    plt.plot(acceleration_norm, label="Acceleration norm")
    plt.xlabel("Frame")
    plt.ylabel("Norm")
    plt.title("Motion Dynamics")
    plt.legend()
    plt.savefig("debug_motion_dynamics.png")
    plt.close()
    return velocity_norm, acceleration_norm

def motion_warning(velocity_norm, acceleration_norm, threshold=1e-3):
    v_mean = np.mean(velocity_norm)
    a_mean = np.mean(acceleration_norm)
    print(f"🔍 Mean velocity: {v_mean:.6f}, Mean acceleration: {a_mean:.6f}")
    if v_mean < threshold and a_mean < threshold:
        print("🚨 WARNING: Output motion is nearly static.")
    else:
        print("✅ Motion dynamics: Reasonable.")

def main():
    skeleton_csv_path = "./output/predicted_skeleton.csv"
    scaler_path = "./scaler/skeleton_scaler.pkl"

    if not os.path.exists(skeleton_csv_path):
        print(f"❌ File not found: {skeleton_csv_path}")
        return
    if not os.path.exists(scaler_path):
        print(f"❌ File not found: {scaler_path}")
        return

    sample_data = pd.read_csv(skeleton_csv_path).values[:10]
    check_scaler_consistency(scaler_path, sample_data)
    analyze_skeleton_statistics(skeleton_csv_path)
    v_norm, a_norm = analyze_motion_dynamics(skeleton_csv_path)
    motion_warning(v_norm, a_norm)

if __name__ == "__main__":
    main()
