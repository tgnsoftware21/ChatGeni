from app.routes import route_app
from app.SQLQueryGenerator import generate_sql
from app.ImageReview import generatecomparison


def register_blueprints(app):
    app.register_blueprint(route_app)
    app.register_blueprint(generate_sql)
    app.register_blueprint(generatecomparison)