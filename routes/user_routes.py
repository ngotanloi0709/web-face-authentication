from flask import Blueprint, render_template

user = Blueprint('user', __name__)


@user.route('/')
def index():
    return "User Index Page"


@user.route('/profile')
def profile():
    return "User Profile Page"


@user.route('/settings')
def settings():
    return "User Settings Page"


