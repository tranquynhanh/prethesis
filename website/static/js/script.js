function formatDateTime(Chosenday) {
    const weekday = new Array(7);
    weekday[0] = "Sunday";
    weekday[1] = "Monday";
    weekday[2] = "Tuesday";
    weekday[3] = "Wednesday";
    weekday[4] = "Thursday";
    weekday[5] = "Friday";
    weekday[6] = "Saturday";
  
    let day = weekday[Chosenday.getDay()];
  
    return `${day}`
    // ${Chosenday.getHours()}:${Chosenday.getMinutes()}`;
  }

  let now = new Date()  
  let curTime = formatDateTime(now);
  let day = document.querySelector(".temperature__today p");
  day.innerHTML = curTime;
  let time = document.querySelector(".time");
  time.innerHTML = `${now.getHours()} : ${now.getMinutes()}`;
  
  function formatDate(Chosenday) {
    const months = new Array(12);
    months[0] = "Jan";
    months[1] = "Feb";
    months[2] = "Mar";
    months[3] = "Apr";
    months[4] = "May";
    months[5] = "Jun";
    months[6] = "Jul";
    months[7] = "Aug";
    months[8] = "Sep";
    months[9] = "Oct";
    months[10] = "Nov";
    months[11] = "Dec";
  
    let month = months[Chosenday.getMonth()];
  
    return `${Chosenday.getDate()} ${month}, ${Chosenday.getFullYear()}`;
  }
  
  let curDate = document.querySelector(".date");
  curDate.innerHTML = formatDate(now);

var x = document.querySelector(".temperature__today small");
function getLocation() {
    navigator.geolocation.getCurrentPosition(showPosition);
}

function showPosition(position) {
    x.innerHTML = `${Math.round(position.coords.latitude, 5)}N - ${Math.round(position.coords.longitude, 5)}E`;
}
getLocation()
