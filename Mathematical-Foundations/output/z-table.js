const TABLE = document.querySelectorAll("table.dataframe");

const thead = document.querySelector("thead");
let TH = Array.from(thead.querySelectorAll("th"));
TH = TH.map((e) => e.innerText).filter((e) => !isNaN(e));

TABLE.forEach((table) => {
  const TR = table.querySelectorAll("tbody tr");
  TR.forEach((tr) => {
    const th = tr.querySelector("th").innerText;
    const TD = tr.querySelectorAll("td");
    for (const [i, td] of TD.entries()) {
      const span = document.createElement("span");
      span.className = "tooltip";
      span.innerText += `[${th}, ${TH[i]}]`;
      td.appendChild(span);
    }
  });
});
