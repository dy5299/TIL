const goormModule = function() {
  
};

goormModule.prototype = {
  init: function() {
    this.setEvent();
    this.loadScript();
  },
  
  setEvent: function() {
    document.onclick = function(e) {
      if (e.target.classList.value.indexOf('addBtn') > -1) {
        challenge.addData(document.getElementById('name').value, document.getElementById('description').value);
      } else if (e.target.classList.value.indexOf('removeBtn') > - 1) {
        challenge.removeData(e.target.parentNode.parentNode.attributes.index.value);
      }
    };
  },
  
  loadScript: function() {
    var scriptTag = document.createElement('script'),
    firstScriptTag = document.getElementsByTagName('script')[0];
    scriptTag.src = 'script.js';
    firstScriptTag.parentNode.insertBefore(scriptTag, firstScriptTag);
  }
};