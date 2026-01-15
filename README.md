# Next Word Prediction with LSTM

This project is a Streamlit-based web application that predicts the next word in a sequence using an LSTM model. The application is designed to be deployed on Azure App Service.

## Features
- Predicts the next word in a given sequence of text.
- Uses a pre-trained LSTM model (`next_word_lstm.h5`).
- Includes a tokenizer (`tokenizer.pickle`) for text preprocessing.
- Interactive web interface built with Streamlit.

## Requirements
- Python 3.10 or higher
- Azure App Service compatible

### Dependencies
The required dependencies are listed in `requirements.txt`. Key dependencies include:
- `tensorflow`
- `streamlit`
- `numpy`
- `pandas`
- `scikit-learn`
- `pickle`

## Deployment on Azure

### Steps to Deploy
1. **Prepare the Application**:
   - Ensure `application.py`, `next_word_lstm.h5`, and `tokenizer.pickle` are in the root directory.
   - Verify that `requirements.txt` includes all necessary dependencies.

2. **Create a Procfile**:
   - The `Procfile` specifies the command to run the Streamlit app:
     ```
     web: streamlit run application.py --server.port $PORT --server.address 0.0.0.0
     ```

3. **Deploy to Azure**:
   - Use the Azure CLI or Azure Portal to create an App Service.
   - Deploy the application files to the App Service.
   - Ensure the App Service is configured to use Python 3.10 or higher.

4. **Test the Deployment**:
   - Access the deployed application via the Azure App Service URL.
   - Verify that the application runs correctly and predicts the next word.

## Notes
- The application is compatible with Azure App Service.
- Ensure that the `next_word_lstm.h5` and `tokenizer.pickle` files are included in the deployment package.
- The `Procfile` is configured for Streamlit to run on Azure.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Author
Developed by [Your Name].