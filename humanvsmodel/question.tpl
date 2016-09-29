<html>
<head>
<script>
function getCursorPosition(event) {
	var img = document.getElementById("image") 
	var indication = document.getElementById("rect") 
	
    var rect = img.getBoundingClientRect();
	indication.style.top = Math.round(event.clientY - 5);
	indication.style.left = Math.round(event.clientX - 5);
	indication.style.display = "block";
    var x = Math.round(event.clientX) - rect.left;
    var y = Math.round(event.clientY) - rect.top;
	indication.style.top = Math.round(y - 5) + 100;
	indication.style.left = Math.round(x - 5) + 10;

	var wart_y = document.getElementById("wart_y");
	wart_y.value = y;
	var wart_x = document.getElementById("wart_x");
	wart_x.value = x;
}

function removeRect(event){
	var wart_y = document.getElementById("wart_y");
	wart_y.value = "";
	var wart_x = document.getElementById("wart_x");
	wart_x.value = "";

	var indication = document.getElementById("rect") 
	indication.style.display = "none";
}

function check(event){
	if ({{idx}} == 1) {
		var wart_y = document.getElementById("wart_y");
		if (wart_y.value == "") {
			event.preventDefault();
			alert('Please click on the center of the wart in the image!');
		}
	}
}

window.ondragstart = function() { return false; } 

</script>
</head>
<body>
<h1>Question {{idx}} / 50</h1>
<img id="image" style="position:absolute; top:100px; left:10px;" src="/images/{{idx}}.png" onclick="getCursorPosition(event)" />
<div id="rect" style="border: 3px solid #000; height: 10px; width: 10px; position:absolute; pointer-events:none; display:none; border-radius:10px;"></div>
<form action="/save" method="post" style="padding-top: 550px">
<h2>Please click on the *center* of the pictured wart (if you see one)</h2>

<a href="#!" style="color:blue" onclick="removeRect(event)">Remove selection on picture</a><br/>
<input type="hidden" name="q_idx" value="{{idx}}"></input> <br />
<input type="hidden" name="wart_x" value="" id="wart_x"></input> <br />
<input type="hidden" name="wart_y" value="" id="wart_y"></input> <br />
<input type="radio" name="type" value="wart" id='wart'> <label for="wart">A wart/skin lesion <strong>without cream</strong></label> <br /><br />
<input type="radio" name="type" value="cream" id='cream'> <label for="cream">There is <strong>cream</strong>.</label> <br /><br />
<input type="radio" name="type" value="none" id='none'> <label for="none">There is no skin lesion and no cream. <br />
<br />
<br />
<br />
<input type="submit" value="Next" onclick="check(event)"></input>
</form>
</body>
