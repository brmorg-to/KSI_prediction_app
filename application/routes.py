from application import app, views

app.add_url_rule('/', methods=['GET', 'POST'], view_func=views.index)
app.add_url_rule('/home', methods=['GET', 'POST'], view_func=views.index)
app.add_url_rule('/prediction', methods=['GET', 'POST'], view_func=views.predict_one)
app.add_url_rule('/predictions', methods=['GET', 'POST'], view_func=views.predict_many)