var login_button= document.getElementById("login");
var login_form=document.getElementById("wrapper");

login_button.addEventListener('click',function(){
    login_form.style.display= 'block';
    login_form.style.animate= 'block';

})


var sidebarVisible = false;

function toggleSidebar() {
  var sidebar = document.getElementById("sidebar");
  var content = document.getElementById("content");
  var page = document.getElementsByTagName("body");

  sidebarVisible = !sidebarVisible;

  if (sidebarVisible) {
    sidebar.style.display = "block";
    content.style.marginLeft = "200px";
   

  } else {
    sidebar.style.display = "none";
    content.style.marginLeft = "0";
  }
}


    var sidebarVisible = true;
    
    function toggleSidebar2() {
      var sidebar = document.getElementById("sidebar");
      var content = document.getElementById("content");
    
      sidebarVisible = !sidebarVisible;
    
      if (sidebarVisible) {
        sidebar.style.display = "block";
        content.style.marginLeft = "200px";
      } else {
        sidebar.style.display = "none";
        content.style.marginLeft = "0";
      }
    }
    
    window.addEventListener('resize', function() {
var screenWidth = window.innerWidth;
var sidebar = document.getElementById("sidebar");
var content = document.getElementById("content");

if (screenWidth > 768) { // Change 768 to your desired breakpoint
sidebar.style.display = "none";
content.style.marginLeft = "0";
}
});


    
