from recaptcha_classifier import (
    DetectionLabels,
    DataPreprocessingPipeline,
    MainCNN,
    Trainer,
    evaluate_model
)
import torch


def main():
    pipeline = DataPreprocessingPipeline(
        DetectionLabels,
        balance=True
        )

    loaders = pipeline.run()
    
    model = MainCNN(
        n_layers=2,#3,
        kernel_size=3,
        num_classes=len(DetectionLabels),
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trainer = Trainer(
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        epochs=10,
        optimizer=optimizer,
        scheduler=scheduler,
        save_folder='models',
        device=device
    )
    
    trainer.train(model)
    
    history = trainer.loss_acc_history
    print("Training completed. Loss and accuracy history:")
    print(history)
    
    results = evaluate_model(
        model=model,
        test_loader=loaders['test'],
        device=device,
        num_classes=len(DetectionLabels),
        class_names=DetectionLabels.dataset_classnames(),
        plot_cm=True
    )
    
    print("Evaluation results:")
    for key, value in results.items():
        print(f"{key}: {value}")

if __name__ == '__main__':
    main()
