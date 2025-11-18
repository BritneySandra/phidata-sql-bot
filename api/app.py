from phi.app import App
from agent import agent

app = App(agent=agent)

if __name__ == "__main__":
    app.run()
