def train_simple_cnn():
    from recaptcha_classifier.pipeline.simple_cnn_pipeline import SimpleClassifierPipeline

    pipeline = SimpleClassifierPipeline()
    pipeline.run(save_train_checkpoints=False)
    
def train_main_classifier():
    from recaptcha_classifier.pipeline.main_model_pipeline import MainClassifierPipeline

    pipeline = MainClassifierPipeline(epochs=1, k_folds=2)
    pipeline.run(save_train_checkpoints=False)
    
def open_api():
    import uvicorn
    uvicorn.run("recaptcha_classifier.api:app", host="0.0.0.0", port=8000, reload=True, workers=1)

def ui():
    from recaptcha_classifier.ui import StreamlitApp

    app = StreamlitApp()
    app.render()

def main():
    # train_simple_cnn()
    # train_main_classifier()
    # open_api()
    ui()


if __name__ == '__main__':
    main()