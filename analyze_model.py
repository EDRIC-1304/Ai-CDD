import os
import torch
import pickle
import sys
from pathlib import Path

def analyze_model_file(file_path: str):
    """
    Analyze the model file to understand its format and structure
    """
    print(f"🔍 Analyzing model file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    # Check file size
    file_size = os.path.getsize(file_path)
    print(f"📁 File size: {file_size / (1024*1024):.2f} MB")
    
    # Try different loading methods
    methods = [
        ("PyTorch State Dict", lambda: torch.load(file_path, map_location='cpu')),
        ("Pickle", lambda: pickle.load(open(file_path, 'rb'))),
    ]
    
    for method_name, load_func in methods:
        try:
            print(f"\n🔄 Trying {method_name}...")
            data = load_func()
            
            print(f"✅ Successfully loaded with {method_name}")
            
            # Analyze the loaded data
            if isinstance(data, dict):
                print(f"📋 Data type: Dictionary")
                print(f"🔑 Keys: {list(data.keys())}")
                
                if 'state_dict' in data:
                    print("🎯 Contains state_dict - full model saved")
                    analyze_state_dict(data['state_dict'])
                elif all(k.startswith('.') or k.replace('.', '').replace('_','').isalnum() for k in data.keys()):
                    print("🏗️  Appears to be a raw state_dict")
                    analyze_state_dict(data)
                else:
                    print("❓ Unknown dictionary structure")
                    
            elif hasattr(data, 'state_dict'):
                print("🎯 Contains model object with state_dict")
                analyze_state_dict(data.state_dict())
            else:
                print(f"❓ Unknown data type: {type(data)}")
            
            return data
            
        except Exception as e:
            print(f"❌ {method_name} failed: {e}")
            continue
    
    print("❌ Could not load file with any method")
    return None

def analyze_state_dict(state_dict):
    """Analyze PyTorch state dict to understand model architecture"""
    print(f"\n🏗️  Model Architecture Analysis:")
    print(f"📊 Total parameters: {len(state_dict)}")
    
    # Categorize parameters
    categories = {
        'vision_transformer': [],
        'moe_layer': [],
        'classifier': [],
        'other': []
    }
    
    for key in state_dict.keys():
        if any(x in key.lower() for x in ['patch', 'pos', 'block', 'norm', 'attn', 'mlp']):
            categories['vision_transformer'].append(key)
        elif any(x in key.lower() for x in ['router', 'expert']):
            categories['moe_layer'].append(key)
        elif 'fc' in key.lower():
            categories['classifier'].append(key)
        else:
            categories['other'].append(key)
    
    for category, params in categories.items():
        if params:
            print(f"  📂 {category.replace('_', ' ').title()}: {len(params)} parameters")
            for param in params[:3]:  # Show first 3
                shape = state_dict[param].shape
                print(f"    - {param}: {shape}")
            if len(params) > 3:
                print(f"    ... and {len(params) - 3} more")

def find_model_files():
    """Find model files in the project"""
    print("🔍 Searching for model files...")
    
    model_extensions = ['.pth', '.pt', '.pkl', '.adding']
    found_files = []
    
    for root, dirs, files in os.walk('.'):
        for file in files:
            if any(file.endswith(ext) for ext in model_extensions):
                found_files.append(os.path.join(root, file))
    
    if found_files:
        print(f"\n📁 Found {len(found_files)} model file(s):")
        for file in found_files:
            size = os.path.getsize(file) / (1024*1024)
            print(f"  - {file} ({size:.2f} MB)")
    else:
        print("❌ No model files found")
    
    return found_files

def main():
    """Main function to analyze model files"""
    print("🏥 Lung Disease Model Analyzer")
    print("=" * 50)
    
    # First, try to find the specific file
    target_file = "lung.adding"
    
    if os.path.exists(target_file):
        print(f"🎯 Found target file: {target_file}")
        analyze_model_file(target_file)
    else:
        print(f"❌ Target file '{target_file}' not found")
        
        # Search for other model files
        found_files = find_model_files()
        
        if found_files:
            print(f"\n🤔 Did you mean one of these files?")
            for i, file in enumerate(found_files, 1):
                print(f"  {i}. {file}")
            
            try:
                choice = input(f"\nEnter file number (1-{len(found_files)}) or file path: ").strip()
                
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(found_files):
                        analyze_model_file(found_files[idx])
                else:
                    analyze_model_file(choice)
            except (ValueError, IndexError):
                print("❌ Invalid choice")
        else:
            print("\n💡 To use your trained model:")
            print("1. Place your model file in the project directory")
            print("2. Make sure it's one of: .pth, .pt, .pkl, .adding")
            print("3. Run: python src/training/predict_trained.py your_model_file image.jpg")

if __name__ == "__main__":
    main()
