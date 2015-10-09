from setuptools import setup

config = {
    'description': 'Web app that allows you to check how positive someone is on Twitter',
    'author': 'mromnia',
    'url': 'https://github.com/mromnia/positivebird',
    'author_email': 'mr.omnia.dev@gmail.com',
    'version': '0.1',
    'install_requires': ['flask', 'twitter', 'nltk', 'psycopg2', 'requests'],
    'packages': ['positivebird'],
    'name': 'positivebird'
}

setup(**config)