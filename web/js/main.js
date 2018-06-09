$(document).ready(function() {
  util.resizeDivHeightToScreen('main__container');
});
$(window).resize(function() {
  util.resizeDivHeightToScreen('main__container');
});

$('.sel').click(function(e) {
    e.preventDefault();
    $('.timelapse').attr('src', 'res/dev_' + this.text + '.gif');
    $('.preview').attr('src', 'res/out_blend_' + this.text + '.png');
});
