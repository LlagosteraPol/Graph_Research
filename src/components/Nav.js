import React from 'react';
import { Link } from 'gatsby';

export default function Nav({ onMenuToggle = () => {} }) {
  return (
    <nav id="nav">
      <ul>
        <li className="special">
          <a
            href="#menu"
            onClick={e => {
              e.preventDefault();
              onMenuToggle();
            }}
            className="menuToggle"
          >
            <span>Menu</span>
          </a>
          <div id="menu">
            <ul>
              <li>
                <Link to="/">Home</Link>
              </li>
              <li>
                <Link to="/Generic">Generic Page</Link>
              </li>
              <li>
                <Link to="/Elements">Elements</Link>
              </li>
              <li>
                <Link to="#volunteering">Voluntariado</Link>
              </li>
              <li>
                <Link to="#projects">Proyectos</Link>
              </li>
              <li>
                <Link to="#associations">Asociaciones</Link>
              </li>
              <li>
                <Link to="#three">La plataforma</Link>
              </li>
              <li>
                <Link to="#cta">Contactar</Link>
              </li>
            </ul>
            <a
              className="close"
              onClick={e => {
                e.preventDefault();
                onMenuToggle();
              }}
              href="#menu"
            >
              {''}
            </a>
          </div>
        </li>
      </ul>
    </nav>
  );
}
