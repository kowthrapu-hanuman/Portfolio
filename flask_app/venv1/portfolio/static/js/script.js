document.addEventListener("DOMContentLoaded", function() {
  const text = "👋 Hi, I'm Kowthrapu Hanuman — A Backend Developer.";
  const typingElement = document.getElementById("typing-text");
  let index = 0;

  function type() {
    if (index < text.length) {
      typingElement.innerHTML = text.substring(0, index + 1);
      index++;
      setTimeout(type, 100);
    }
  }

  type();
});
