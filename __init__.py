from flask import Flask, render_template, request
# Inside GitHubScrape/app/__init__.py
from summarize import summarize_text
from scraper import get_readme_content


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    github_url = request.form.get('github_url')
    
    # Retrieve README content from the GitHub URL
    readme_content ,links= get_readme_content(github_url)
    
    if readme_content:
        # Your Hugging Face access token
        token = "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"

        # Model name
        model_name = "LaMini-Flan-T5-248M"  # Replace with your model name
        
        try:
            # Summarize the README content
            summary = summarize_text(readme_content, model_name, token)
            return render_template('index.html', summary=summary)
        except Exception as e:
            error_message = f"Error summarizing content: {e}"
            return render_template('index.html', error=error_message)
    else:
        error_message = "Failed to retrieve README content. Please check the URL."
        return render_template('index.html', error=error_message)

if __name__ == '__main__':
    app.run(debug=True)
