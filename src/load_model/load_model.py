import torch
import torch.nn as nn
from pathlib import Path

def load_and_inspect_model():
    """Lädt das Modell und gibt die Struktur aus."""

    # Modell-Pfad (relativ vom Script-Verzeichnis)
    model_path = Path("models/moabb_downsampled_good_subjects_model_full.pth")

    if not model_path.exists():
        print(f"❌ Modell nicht gefunden: {model_path}")
        print(f"Absoluter Pfad: {model_path.absolute()}")
        return None

    try:
        # Modell laden
        print(f"📦 Lade Modell: {model_path}")
        model = torch.load(model_path, map_location='cpu', weights_only=False)

        print("\n" + "="*60)
        print("📋 MODELL-STRUKTUR")
        print("="*60)

        # Typ des geladenen Objekts
        print(f"Typ: {type(model)}")

        # Falls es ein Dictionary ist (state_dict + metadata)
        if isinstance(model, dict):
            print(f"\nDict Keys: {list(model.keys())}")

            # State dict analysieren
            if 'state_dict' in model:
                print("\n🧠 STATE_DICT LAYERS:")
                for name, param in model['state_dict'].items():
                    print(f"  {name}: {param.shape}")

            # Weitere Metadaten
            for key, value in model.items():
                if key != 'state_dict':
                    print(f"\n{key.upper()}: {value}")

        # Falls es direkt ein nn.Module ist
        elif isinstance(model, nn.Module):
            print("\n🧠 MODELL-ARCHITEKTUR:")
            print(model)
            # Zugriff auf das letzte Layer der classifier-Sequential (falls vorhanden)
            device = next(model.parameters()).device  # Get model device
            x = torch.randn(1, 16, 500, device=device)  # Dummy-Input auf richtiges Device
            with torch.no_grad():
                y = model(x)
            print(y.shape)  # sollte [1, 4, 12] sein

            print("\n📏 PARAMETER:")
            total_params = 0
            for name, param in model.named_parameters():
                print(f"  {name}: {param.shape} ({param.numel()} params)")
                total_params += param.numel()

            print(f"\n📊 GESAMT PARAMETER: {total_params:,}")

            # Model summary
            print(f"\nTraining Mode: {model.training}")

        else:
            print(f"Unbekannter Modell-Typ: {type(model)}")
            print(f"Inhalt: {model}")

        print("\n" + "="*60)
        return model

    except Exception as e:
        print(f"❌ Fehler beim Laden: {e}")
        return None

if __name__ == "__main__":
    model = load_and_inspect_model()