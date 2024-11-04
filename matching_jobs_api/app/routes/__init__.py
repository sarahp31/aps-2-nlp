def register_routes(app):
    from app.routes.index import index_bp

    app.register_blueprint(index_bp)