document.getElementById("rentalForm").addEventListener("submit", function(event) {
  let formValid = true;

  function validateNumberField(inputName, errorId, errorMessage) {
    const input = document.querySelector("input[name='" + inputName + "']");
    const error = document.getElementById(errorId);
    if (isNaN(input.value) || input.value === "") {
      input.classList.add("error");
      error.style.display = "block";
      error.textContent = errorMessage;
      formValid = false;
    } else {
      input.classList.remove("error");
      error.style.display = "none";
    }
  }

  validateNumberField("area", "areaError", "אנא הזן מספר חוקי לשטח.");
  validateNumberField("garden_area", "gardenAreaError", "יש להזין ערך מספרי או 0.");

  // Floor vs total floors
  const floorInput = document.querySelector("input[name='floor']");
  const totalFloorsInput = document.querySelector("input[name='total_floors']");
  if (parseInt(floorInput.value) > parseInt(totalFloorsInput.value)) {
    floorInput.classList.add("error");
    totalFloorsInput.classList.add("error");
    formValid = false;
    if (!document.getElementById("floorTotalError")) {
      const errorMsg = document.createElement("div");
      errorMsg.id = "floorTotalError";
      errorMsg.className = "error-message";
      errorMsg.textContent = "קומה לא יכולה להיות גבוהה ממספר הקומות בבניין.";
      floorInput.parentNode.insertBefore(errorMsg, floorInput.nextSibling);
    }
    document.getElementById("floorTotalError").style.display = "block";
  } else {
    floorInput.classList.remove("error");
    totalFloorsInput.classList.remove("error");
    const floorTotalError = document.getElementById("floorTotalError");
    if (floorTotalError) floorTotalError.style.display = "none";
  }

  // Neighborhood selection
  const neighborhoodSelect = document.querySelector("select[name='neighborhood']");
  if (neighborhoodSelect.value === "") {
    neighborhoodSelect.classList.add("error");
    if (!document.getElementById("neighborhoodError")) {
      const error = document.createElement("div");
      error.id = "neighborhoodError";
      error.className = "error-message";
      error.textContent = "אנא בחר שכונה מתוך הרשימה.";
      neighborhoodSelect.parentNode.insertBefore(error, neighborhoodSelect.nextSibling);
    } else {
      document.getElementById("neighborhoodError").style.display = "block";
    }
    formValid = false;
  } else {
    neighborhoodSelect.classList.remove("error");
    const error = document.getElementById("neighborhoodError");
    if (error) error.style.display = "none";
  }

  // Property type selection
  const propertyTypes = document.querySelectorAll("input[name='property_type']");
  const selected = Array.from(propertyTypes).some(r => r.checked);
  if (!selected) {
    propertyTypes.forEach(el => el.classList.add("error"));
    if (!document.getElementById("propertyTypeError")) {
      const error = document.createElement("div");
      error.id = "propertyTypeError";
      error.className = "error-message";
      error.textContent = "אנא בחר סוג נכס.";
      propertyTypes[propertyTypes.length - 1].parentNode.parentNode.appendChild(error);
    } else {
      document.getElementById("propertyTypeError").style.display = "block";
    }
    formValid = false;
  } else {
    propertyTypes.forEach(el => el.classList.remove("error"));
    const error = document.getElementById("propertyTypeError");
    if (error) error.style.display = "none";
  }

  if (!formValid) {
    event.preventDefault();
  }
});
