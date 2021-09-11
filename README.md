# GoodyOrBaddy
An experiment in Machine Learning image classification using Microsoft ML.NET from an Azure Function

## Project Setup

### Locally
In order to get the site runing locally you will have to:

#### Step 1 Create the ML model

Uncomment the CreateModel() method in the Console Application Program.cs and created the model-v1.zip file. 

Note: Make sure all the paths in the code are correct first.

Note: When running this console application it may take a while and you may see message suggesting something is wrong.. unless they are .NET exceptions and can be ignored. Wait patiently and the model will be created.

#### Step 2 Configure your local Azure Function

Add a CORS setting to your Azure Functions local.settings.json file that allows your local site to call it

#### Step 3 Configure your local front end web page

Update index.html so that the url property of the dropzone.js is your locally running Azure Function

#### Step 4 Serve the front end web page using a local server locally

Serve the index.html file from some sort of local server (for example Visual Studio Codes 'Live Server' plugin)


### Online
In order to get the site running online, you will have to: 

- Publish the Azure Function to your Azure instance

- Update index.html so that dropzone.js sends the fike to your live Azure Function.

- Add a CORS value in your Azure Function settings in your Azure Portal that allows your domain sending the image.

- Publish your index.html page somewhere


## Project Components

### docs 
This contains a simple front end html page that uses dropzone.js to send images to the Azure function and displays the results. The html page is actually hosted as GitHub page.

### ImageClassification.AzureFunction
This Azure function that receives the image and runs it through the Machine Learning model and returns the result

### ImageClassification.Library
A simple class library that stores some POCO classes that are used by the Azure Function and Console Application

### ImageClassification.CoreConsoleApplication
This Console Application creates the image classification model.
