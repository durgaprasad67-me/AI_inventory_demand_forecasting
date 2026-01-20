<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Powered Inventory Demand Forecasting</title>
    <link rel="icon" type="image/png" href="logo.png">

    <style>
        body {
            background-color: rgb(50, 162, 139);
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }

        .box {
            border: 2px dashed white;
            border-radius: 25px;
            width: 250px;
            height: 250px;
            background-color: rgb(246, 251, 252);

            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;

            cursor: pointer;
        }

        .box img {
            width: 180px;
        }

        .upload-text {
            margin-top: 10px;
            font-size: 14px;
            color: black;
            font-family: Arial, sans-serif;
        }

        .file-name {
            margin-top: 6px;
            font-size: 13px;
            word-break: break-all;
            text-align: center;
        }

        .Clear{
            border: 2px dashed white;
            border-radius:15px;
            width:50px;
            height:20px;
            background-color: rgb(246, 251, 252);

            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;

            cursor: pointer;
        }
        
    </style>
</head>
<body>

<form action="/upload" method="POST" enctype="multipart/form-data">
    <br><label class="box">
        <img src="Upload_icon.png" alt="Upload CSV">

        <p class="upload-text" id="uploadText">
            Upload a CSV file
        </p>

        <p class="file-name" id="fileName"></p>

        <input 
            type="file" 
            name="file" 
            accept=".csv"
            hidden
            onchange="showFileName(this)"
        >
    </label></br>
    
   <div class="text-center mt-8">
      <a href="index.html"
         class="Clear"> 
         Clear
      </a>
    </div>
</form>

<script>
function showFileName(input) {
    const fileNameText = document.getElementById("fileName");
    const uploadText = document.getElementById("uploadText");

    if (input.files.length === 0) return;

    const file = input.files[0];
    const name = file.name.toLowerCase();

    if (!name.endsWith(".csv")) {
        fileNameText.textContent = "The file isn't .csv";
        fileNameText.style.color = "red";
        uploadText.textContent = "Upload a CSV file";
        input.value = "";
        return;
    }

    fileNameText.textContent = file.name;
    fileNameText.style.color = "green";
    uploadText.textContent = "Selected file:";
}

</script>



</body>
</html>
