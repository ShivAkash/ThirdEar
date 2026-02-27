import sys
import os

# Create pos/neg dirs
os.makedirs("test_data/pos", exist_ok=True)
os.makedirs("test_data/neg", exist_ok=True)

if "piper-sample-generator/" not in sys.path:
    sys.path.append("piper-sample-generator/")

from generate_samples import generate_samples

print("Generating POSITIVE samples (shiv_aak_aash)")
generate_samples(text="shiv_aak_aash",
                 max_samples=5,
                 length_scales=[1.0, 1.1, 1.2],
                 noise_scales=[0.3, 0.5, 0.7], 
                 noise_scale_ws=[0.3, 0.5, 0.7],
                 output_dir="./test_data/pos", 
                 batch_size=1, 
                 auto_reduce_batch_size=True,
                 file_names=[f"pos_{i}.wav" for i in range(5)])

print("Generating NEGATIVE samples (random sentences)")
generate_samples(text="hello how are you doing today",
                 max_samples=2,
                 output_dir="./test_data/neg", 
                 batch_size=1, 
                 auto_reduce_batch_size=True,
                 file_names=["neg_0.wav", "neg_1.wav"])

generate_samples(text="the weather is very nice outside",
                 max_samples=2,
                 output_dir="./test_data/neg", 
                 batch_size=1, 
                 auto_reduce_batch_size=True,
                 file_names=["neg_2.wav", "neg_3.wav"])

generate_samples(text="play some music please",
                 max_samples=1,
                 output_dir="./test_data/neg", 
                 batch_size=1, 
                 auto_reduce_batch_size=True,
                 file_names=["neg_4.wav"])

print("Done generating test samples.")
