console.log("Moob Admin");

ApplicationLoad = {
  init: function () {
    this.load_page();
  },

  load_page: function(){
    $(window).load(function(){
      $("#load").fadeOut(500).delay(500);
    });

  },

}

$(document).ready(function(){ 
	ApplicationLoad.init();


$('.tooltips').tooltip();

$(".logout").on('click', function(event) {
  event.preventDefault();

  loadIn();

  dados = {"logout": "true"};

  jQuery.ajax({
        type: "POST",
        url: "inc/modules/usuario/logout.php",
        data: dados,
        success: function(data)
        { 
          $(".ajax-html").html(data);
          loadOut();
        }
      });


  });

$(function () {
    $('.navbar-toggle').click(function () {
        $('.navbar-nav').toggleClass('slide-in');
        $('.side-body').toggleClass('body-slide-in');
        $('#search').removeClass('in').addClass('collapse').slideUp(200);

        /// uncomment code for absolute positioning tweek see top comment in css
        //$('.absolute-wrapper').toggleClass('slide-in');
        
    });
   
   // Remove menu for searching
   $('#search-trigger').click(function () {
        $('.navbar-nav').removeClass('slide-in');
        $('.side-body').removeClass('body-slide-in');

        /// uncomment code for absolute positioning tweek see top comment in css
        //$('.absolute-wrapper').removeClass('slide-in');

    });
});



});

function mensagem(html){
		$('.ajax-html').html(html);
		$('#modal').modal('show');
}

function loadIn(){
	$("#load").fadeIn(500).delay(500);
}
function loadOut(){
	$("#load").fadeOut(500).delay(500);
}

function imageUpload(file, largura, altura){


}

