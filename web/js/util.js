var util = (function() {
  var resizeDivHeightToScreen = function(id) {
    if (!id.startsWith('#'))
      id = '#' + id;
    var $div = $(id);
    $div.outerHeight($(window).height());
  };
  var util = {
    resizeDivHeightToScreen: resizeDivHeightToScreen,
  };
  return util;
})();
