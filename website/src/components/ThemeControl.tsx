import { Switch } from "antd";
import { useEffect, useState } from "react";

type Props = {
  darkLabel: string;
  lightLabel: string;
};

export default function ThemeControl({ darkLabel, lightLabel }: Props) {
  const [dark, setDark] = useState(false);

  useEffect(() => {
    const saved = window.localStorage.getItem("typenn-theme");
    const initial = saved === "dark" || (saved !== "light" && window.matchMedia("(prefers-color-scheme: dark)").matches);
    setDark(initial);
    document.documentElement.dataset.theme = initial ? "dark" : "light";
    window.dispatchEvent(new Event("typenn-theme-change"));
  }, []);

  function changeTheme(checked: boolean) {
    setDark(checked);
    document.documentElement.dataset.theme = checked ? "dark" : "light";
    window.localStorage.setItem("typenn-theme", checked ? "dark" : "light");
    window.dispatchEvent(new Event("typenn-theme-change"));
  }

  return (
    <span className="theme-control inline-flex items-center gap-2">
      <Switch
        checked={dark}
        onChange={changeTheme}
        aria-label={dark ? lightLabel : darkLabel}
        size="small"
      />
    </span>
  );
}
