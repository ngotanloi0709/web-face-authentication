from flask import Flask, request

from routes.home_routes import home

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_url_path='/static', static_folder='static')

# Register for home route
app.register_blueprint(home, url_prefix='/')

# App config
app.config['SECRET_KEY'] = 'secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['FACES_FOLDER'] = 'faces'
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.use_reloader = True


# Remove jinja2 cache
@app.before_request
def clear_jinja_cache():
	if 'localhost' in request.host_url or '0.0.0.0' in request.host_url:
		app.jinja_env.cache = {}


# @app.route('/login', methods=['GET', 'POST'])
# @app.route('/', methods=['GET'])
# def login():
# 	if request.method == 'POST':
#
#
# 	return render_template('login.html')


# run project
if __name__ == '__main__':
	app.run(extra_dirs=['uploads', 'faces'])
