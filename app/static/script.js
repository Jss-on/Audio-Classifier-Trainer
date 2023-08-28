function addClass() {
    var container = document.getElementById("class-container");
    var index = container.getElementsByClassName("class-section").length;
    var div = document.createElement("div");
    div.className = "class-section";

    var label1 = document.createElement("label");
    label1.htmlFor = "class_name" + index;
    label1.innerText = "Class " + index + " Name: ";
    div.appendChild(label1);

    var input1 = document.createElement("input");
    input1.type = "text";
    input1.name = "class_name" + index;
    input1.required = true;
    div.appendChild(input1);

    var label2 = document.createElement("label");
    label2.htmlFor = "class" + index;
    label2.innerText = " ZIP File: ";
    div.appendChild(label2);

    var input2 = document.createElement("input");
    input2.type = "file";
    input2.name = "class" + index;
    input2.accept = ".zip";
    input2.required = true;
    div.appendChild(input2);

    container.appendChild(div);
}
