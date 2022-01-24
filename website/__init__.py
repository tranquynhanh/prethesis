from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler


def create_app():
    app = Flask(__name__)
    sched = BackgroundScheduler(daemon=True)
    sched.add_job(sensor,'interval',minutes=60)
    sched.start()

    return app
