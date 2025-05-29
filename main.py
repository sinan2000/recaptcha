import sys 
import questionary


def main():
    """
    App main entry point. If argument is given, it will be used as an action.
    Else, display a menu to select an action.
    """
    args = sys.argv[1:]
    if args:
        handle_action(args[0])
    else:
        choice = select_action()
        handle_action(choice)

def select_action():
    """
    Displays a menu to select what to do with the app.
    """
    return questionary.select("Select an action:", choices=[
        "Open Streamlit UI",
        "Train simple CNN",
        "Train main CNN",
        "Expose API endpoint"
    ]).ask()

def handle_action(choice: str):
    """
    Main action handler. Maps the choice to its
    corresponding logic and executes it.
    """
    action_map = {
        "Open Streamlit UI": ui,
        "--streamlit": ui,
        "Train simple CNN": train_simple_cnn,
        "--train-simple-cnn": train_simple_cnn,
        "Train main CNN": train_main_classifier,
        "--train-main-cnn": train_main_classifier,
        "Expose API endpoint": open_api,
        "--api": open_api
    }
    action = action_map.get(choice)
    if action:
        action()
    else:
        print(f"Unknown argument: {choice}. Please see the menu:")
        choice = select_action()
        handle_action(choice)

def ui():
    from recaptcha_classifier.server.app import StreamlitApp

    app = StreamlitApp()
    app.render()

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
    # opens endpoint at http://localhost:8000/
    uvicorn.run("recaptcha_classifier.server.api:app", reload=True)


if __name__ == '__main__':
    main()