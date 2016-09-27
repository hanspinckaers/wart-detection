<html>
<head>
<script>
function getCursorPosition(event) {
	var img = document.getElementById("image") 
	var indication = document.getElementById("rect") 
	
    var rect = img.getBoundingClientRect();
	indication.style.top = Math.round(event.clientY - 75);
	indication.style.left = Math.round(event.clientX - 75);
	indication.style.display = "block";
    var x = Math.round(event.clientX) - rect.left;
    var y = Math.round(event.clientY) - rect.top;

	var wart_y = document.getElementById("wart_y");
	wart_y.value = y;
	var wart_x = document.getElementById("wart_x");
	wart_x.value = x;
}

window.ondragstart = function() { return false; } 

</script>
</head>
<body>
<h1>Question {{idx}} / 30</h1>
<p>Please click on the pictured wart (if you see one)</p>
<img id="image" style="position:absolute; top:100px;" src="/img/{{idx}}.png" onclick="getCursorPosition(event)" />
<div id="rect" style="border: 2px solid #000; height: 150px; width: 150px; position:absolute; pointer-events:none; display:none;"></div>
<form action="/save" method="post" style="padding-top: 500px">
<input type="hidden" name="q_idx" value="{{idx}}"></input> <br />
<input type="hidden" name="wart_x" value="" id="wart_x"></input> <br />
<input type="hidden" name="wart_y" value="" id="wart_y"></input> <br />
<input type="radio" name="type" value="wart" id='wart'> <label for="wart">A wart <strong>without</strong> cream</label> <br />
<input type="radio" name="type" value="cream" id='cream'> <label for="cream">A wart <strong>with</strong> cream</label> <br />
<br />
<br />
<br />
<input type="submit" value="Submit"></input>
</form>
</body>
