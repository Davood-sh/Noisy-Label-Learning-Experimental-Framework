import numpy as np
import matplotlib.pyplot as plt

def main(history_path: str):
    # Load saved history
    data = np.load(history_path)
    epochs = np.arange(1, data['train_loss'].size + 1)

    # --- Loss (unchanged) ---
    plt.figure(figsize=(8,5))
    plt.plot(epochs, data['train_loss'], label='Train Loss')
    plt.plot(epochs, data['test_loss'],  label='Test Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Train vs. Test Loss')
    plt.legend(); plt.grid(True)
    plt.savefig('loss_curve.png')

    # --- Top-1 Accuracy ---
    plt.figure(figsize=(8,5))
    plt.plot(epochs, data['train_acc_top1'], marker='o', linestyle='-', label='Train Top-1')
    plt.plot(epochs, data['test_acc_top1'],  marker='s', linestyle='--', label='Test Top-1')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('Top-1 Accuracy: Train vs. Test')
    plt.legend(); plt.grid(True)
    plt.savefig('accuracy_top1.png')

    # --- Top-5 Accuracy ---
    plt.figure(figsize=(8,5))
    plt.plot(epochs, data['train_acc_top5'], marker='o', linestyle='-', label='Train Top-5')
    plt.plot(epochs, data['test_acc_top5'],  marker='s', linestyle='--', label='Test Top-5')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('Top-5 Accuracy: Train vs. Test')
    plt.legend(); plt.grid(True)
    plt.savefig('accuracy_top5.png')

    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot training history')
    parser.add_argument('--history', type=str, default='history.npz',
                        help='Path to .npz file with training history')
    args = parser.parse_args()
    main(args.history)
