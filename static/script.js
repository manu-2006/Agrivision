function uploadImage() {
    let file = document.getElementById("imageInput").files[0];

    let formData = new FormData();
    formData.append("file", file);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
        .then(res => res.json())
        .then(data => {
            document.getElementById("result").innerHTML = `
            <h3>${data.disease}</h3>
            <p>Confidence: ${data.confidence}%</p>
            <p>${data.solution}</p>
        `;
        });

    document.getElementById("preview").src = URL.createObjectURL(file);
}