const challenge = {
  /*
   * addData 함수를 완성시켜주세요.
   */
  addData: async function(name, description) {
    // 먼저 서버 쪽에 데이터를 등록합니다.
    //
    fetch('', {
      method: '',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({

      })
    }).then(function(response) {
      return response.json();
    }).then(function(data) {
      // Table에 새 데이터를 넣어 보여줍니다.
      var mytable = document.getElementsByClassName('card-content')[1].getElementsByTagName('table');
      var Row = mytable.insertRow(mytable.rows.length);

      var objCell_Name = Row.insertCell(0);
      objCell_Name.innerHTML = name

      var objCell_Name = Row.insertCell(1);
      objCell_Name.innerHTML = description

      var objCell_Name = Row.insertCell(2);
      objCell_Name.innerHTML = "<a class="removeBtn waves-effect waves-light btn-small">삭제</a>"
      //
    });
  },

  /*
   * removeData 함수를 완성시켜주세요.
   */
  removeData: function(index) {
    // 먼저 서버 쪽에 데이터 삭제 요청을 합니다.
    //
    fetch('', {
      method: '',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({

      })
    }).then(function(response) {
      return response.json();
    }).then(function(data) {
      // Table에서 index 값으로 찾아 데이터를 삭제해줍니다.
      //
    });
  }
}