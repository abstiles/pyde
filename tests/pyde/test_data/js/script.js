(function() {
	"use strict";
	function postLoad() {
		console.log('Page loaded!');
	}
	if (document.readyState === "loading") {
		document.addEventListener("DOMContentLoaded", postLoad);
	} else {
		postLoad();
	}
})();
