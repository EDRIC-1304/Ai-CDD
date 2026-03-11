import torch
import torch.nn as nn
import sys
import os
from typing import Union, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.moet import MoETClassifier

class TrainedModelLoader:
    """Load and use trained models with ViT + MoE architecture"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize trained model loader
        
        Args:
            model_path: Path to trained model file (.pth, .pt, .pkl, or .adding)
            device: Device to load model on ('auto', 'cpu', 'cuda')
        """
        self.model_path = model_path
        self.device = self._get_device(device)
        self.model = None
        
    def _get_device(self, device: str) -> torch.device:
        """Determine the appropriate device"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def load_model(self, model_config: Dict[str, Any] = None) -> MoETClassifier:
        """
        Load trained model with configuration
        
        Args:
            model_config: Model configuration parameters
                          (n_classes, n_experts, top_k)
        
        Returns:
            Loaded MoETClassifier model
        """
        # Default configuration
        default_config = {
            'n_classes': 2,
            'n_experts': 3,
            'top_k': 2
        }
        
        if model_config:
            default_config.update(model_config)
        
        # Initialize model with same architecture as training
        self.model = MoETClassifier(**default_config)
        
        try:
            # Load state dict based on file extension
            if self.model_path.endswith('.adding'):
                # Handle .adding file (might be custom format)
                self.model = self._load_adding_format()
            elif self.model_path.endswith(('.pth', '.pt')):
                # Standard PyTorch format
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            elif self.model_path.endswith('.pkl'):
                # Pickle format
                import pickle
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    if isinstance(model_data, dict) and 'state_dict' in model_data:
                        self.model.load_state_dict(model_data['state_dict'])
                    else:
                        self.model.load_state_dict(model_data)
            else:
                raise ValueError(f"Unsupported model format: {self.model_path}")
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded successfully from {self.model_path}")
            print(f"📱 Device: {self.device}")
            print(f"🏗️  Architecture: ViT + MoE")
            print(f"⚙️  Config: {default_config}")
            
            return self.model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _load_adding_format(self) -> MoETClassifier:
        """
        Handle custom .adding file format
        """
        try:
            # Try different pickle protocols
            import pickle
            
            # Method 1: Try with different protocols
            for protocol in [None, 2, 3, 4, 5]:
                try:
                    with open(self.model_path, 'rb') as f:
                        if protocol is None:
                            data = pickle.load(f)
                        else:
                            data = pickle.load(f, encoding='latin1')
                    
                    print(f"✅ Loaded .adding file with protocol handling")
                    
                    # Handle different data formats
                    if hasattr(data, 'state_dict'):
                        self.model.load_state_dict(data.state_dict())
                    elif isinstance(data, dict):
                        if 'state_dict' in data:
                            self.model.load_state_dict(data['state_dict'])
                        elif all(k.replace('.', '').replace('_','').isalnum() or k.startswith('.') for k in data.keys()):
                            self.model.load_state_dict(data)
                        else:
                            # Try to extract state dict from nested structure
                            for key in data:
                                if isinstance(data[key], dict) and any('weight' in k or 'bias' in k for k in data[key].keys()):
                                    self.model.load_state_dict(data[key])
                                    break
                            else:
                                raise ValueError("Cannot find state dict in .adding file")
                    else:
                        raise ValueError(f"Unknown .adding data type: {type(data)}")
                    
                    return self.model
                    
                except (pickle.UnpicklingError, UnicodeDecodeError, ValueError) as e:
                    print(f"Protocol {protocol} failed: {e}")
                    continue
            
            # Method 2: Try torch.load with different map_locations
            try:
                import torch
                state_dict = torch.load(self.model_path, map_location='cpu', pickle_module=pickle)
                self.model.load_state_dict(state_dict)
                return self.model
            except Exception as e:
                print(f"Torch load failed: {e}")
            
            raise ValueError(f"Cannot load .adding file with any method")
            
        except Exception as e:
            raise ValueError(f"Cannot load .adding file: {e}")
    
    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Make prediction with loaded model
        
        Args:
            input_tensor: Input tensor (batch_size, 3, 224, 224)
        
        Returns:
            Model output tensor
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output = self.model(input_tensor)
            return output
    
    def get_confidence_and_prediction(self, input_tensor: torch.Tensor) -> tuple:
        """
        Get prediction with confidence score
        
        Args:
            input_tensor: Input tensor (batch_size, 3, 224, 224)
        
        Returns:
            Tuple of (prediction, confidence)
        """
        output = self.predict(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probs, dim=1)
        
        return prediction.item(), confidence.item()

# Usage Examples
def example_usage():
    """Example of how to use the trained model loader"""
    
    # Example 1: Load and use trained model
    print("=== Example 1: Basic Usage ===")
    
    # Initialize loader with your trained model
    loader = TrainedModelLoader("lung.adding")  # or "models_saved/tb_moet.pth"
    
    # Load the model
    try:
        model = loader.load_model({
            'n_classes': 2,
            'n_experts': 3,
            'top_k': 2
        })
        
        # Create dummy input for testing
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Make prediction
        prediction, confidence = loader.get_confidence_and_prediction(dummy_input)
        
        result = "NORMAL" if prediction == 0 else "DISEASE DETECTED"
        print(f"🔍 Prediction: {result}")
        print(f"📊 Confidence: {confidence * 100:.2f}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 2: Integration with existing pipeline
    print("\n=== Example 2: Pipeline Integration ===")
    
    def integrate_with_pipeline(model_path: str, image_path: str):
        """Integrate trained model with existing prediction pipeline"""
        import cv2
        
        # Load trained model
        loader = TrainedModelLoader(model_path)
        model = loader.load_model()
        
        # Preprocess image (same as your existing pipeline)
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = torch.tensor(img).permute(2, 0, 1).float()
        img = img.unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        prediction, confidence = loader.get_confidence_and_prediction(img)
        
        result = "NORMAL" if prediction == 0 else "DISEASE DETECTED"
        
        print(f"\n🏥 Medical Diagnosis Result")
        print(f"📷 Image: {image_path}")
        print(f"🔬 Model: {os.path.basename(model_path)}")
        print(f"🎯 Prediction: {result}")
        print(f"📈 Confidence: {confidence * 100:.2f}%")
        
        return prediction, confidence
    
    # Example usage (commented out since we don't have real image)
    # integrate_with_pipeline("lung.adding", "path/to/xray.jpg")

if __name__ == "__main__":
    example_usage()
